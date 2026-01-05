import logging
import os
import asyncio
import glob
import subprocess
import torch
import pysrt
import shutil

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

# --- 📝 DEFAULT PROMPTS ---
DEFAULT_PROMPTS = {
    "burmese": """
Role: Professional Video Narrator & Translator (Burmese).

Strict Translation Guidelines:
1. **Style:** Translate as a **Natural Narrator**. Professional, clear, and engaging.
2. **Multi-Language Support:** The input may contain English, Korean, or other languages. **Translate EVERYTHING into natural Burmese.**
3. **Structure:** - If input is **SRT**, maintain the exact SRT format (timestamps/indexes). ONLY translate the dialogue text.
   - If input is **Text**, just translate naturally.
4. **No Brackets:** NEVER keep original words in brackets.
5. **Abbreviations:** Transliterate phonetically (e.g., VIP → ဗီအိုင်ပီ).
6. **Forbidden:** Do NOT use 'ပေါ့' (pout).
""",
    "rephrase": "Rephrase this text to be more clear, natural, and reliable. If SRT, keep format."
}

# Folders
BASE_FOLDERS = ["downloads", "temp"]
for f in BASE_FOLDERS:
    os.makedirs(f, exist_ok=True)

# User Data
user_prefs = {}
user_modes = {} 
chat_histories = {} 

# --- 🛠️ HELPER FUNCTIONS ---
def get_user_state(user_id):
    if user_id not in user_prefs:
        user_prefs[user_id] = {
            "transcribe_engine": "gemini",
            "output_format": "srt", # Default to SRT
            "custom_prompts": {} 
        }
    return user_prefs[user_id]

def get_active_prompt(user_id, key):
    state = get_user_state(user_id)
    custom = state.get("custom_prompts", {}).get(key)
    return custom if custom else DEFAULT_PROMPTS[key]

def get_paths(user_id):
    # We define base paths, but file extensions logic handles specific srt/txt scenarios
    return {
        "input": f"downloads/{user_id}_input.mp4",
        "audio": f"downloads/{user_id}_audio.mp3",
        "srt": f"downloads/{user_id}_subs.srt",
        "txt": f"downloads/{user_id}_transcript.txt",
        "trans_srt": f"downloads/{user_id}_translated.srt",
        "trans_txt": f"downloads/{user_id}_translated.txt"
    }

def clean_temp(user_id):
    p = get_paths(user_id)
    if os.path.exists(p['input']): os.remove(p['input'])

def wipe_user_data(user_id):
    for f in glob.glob(f"downloads/{user_id}_*"):
        try: os.remove(f)
        except: pass
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

# --- 🧠 ENGINES ---

def run_whisper(audio_path, srt_path, txt_path):
    print(f"🎙️ [Whisper] Processing {audio_path}...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        model = WhisperModel("small", device=device, compute_type=compute_type)
        segments, _ = model.transcribe(audio_path, beam_size=5)
        
        # Always generate both for Whisper locally, it's cheap
        with open(srt_path, "w", encoding="utf-8") as srt, open(txt_path, "w", encoding="utf-8") as txt:
            for i, segment in enumerate(segments, start=1):
                start = format_timestamp(segment.start)
                end = format_timestamp(segment.end)
                text = segment.text.strip()
                srt.write(f"{i}\n{start} --> {end}\n{text}\n\n")
                txt.write(f"{text} ")
        return "Whisper (Local)"
    except Exception as e:
        print(f"Whisper Error: {e}")
        return "Error"

def run_gemini_transcribe(audio_path, output_path, output_format):
    print(f"✨ [Gemini] Listening to {audio_path} (Format: {output_format})...")
    try:
        client = genai.Client(api_key=GEMINI_KEY)
        
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
            
        if output_format == "srt":
            prompt = """
            Transcribe this audio file directly into SRT (SubRip) format.
            1. Use correct timestamp format (00:00:00,000 --> 00:00:00,000).
            2. Break lines naturally for subtitles.
            3. Do not add any markdown, introduction, or conversation. JUST THE SRT CONTENT.
            """
        else:
            prompt = "Transcribe this audio into a clean, readable text transcript. No timestamps. Just text."
        
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3"),
                        types.Part.from_text(text=prompt)
                    ]
                )
            ]
        )
        
        content = response.text.strip()
        # Clean up markdown code blocks if Gemini adds them
        content = content.replace("```srt", "").replace("```", "").strip()
        
        if not content: content = "[No Speech Detected]"
            
        with open(output_path, "w", encoding="utf-8") as f: f.write(content)
        
        return "Gemini 2.0 Flash"
        
    except Exception as e:
        print(f"Gemini Error: {e}")
        with open(output_path, "w") as f: f.write("[Error]")
        return "Error"

def format_timestamp(seconds):
    import math
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{math.floor(seconds):02},{milliseconds:03}"

# --- 🧠 TRANSLATION & CHAT ---

async def run_translate(user_id, prompt_text):
    p = get_paths(user_id)
    
    # Detect Source
    source_path = None
    file_type = "txt"
    
    # Priority: Check what the user last generated/uploaded
    # If they uploaded a file, it saved to p['srt'] or p['txt']
    if os.path.exists(p['srt']) and os.path.getsize(p['srt']) > 0:
        source_path = p['srt']
        file_type = "srt"
        output_path = p['trans_srt']
    elif os.path.exists(p['txt']) and os.path.getsize(p['txt']) > 0:
        source_path = p['txt']
        file_type = "txt"
        output_path = p['trans_txt']
    
    if not source_path: return False, "❌ No content found (SRT or TXT) to translate."

    print(f"🌍 Translating {source_path} ({file_type})...")
    client = genai.Client(api_key=GEMINI_KEY)
    
    with open(source_path, "r", encoding="utf-8") as f: original_text = f.read()
    
    # Specialized Prompt for SRT vs TXT
    if file_type == "srt":
        system_instruction = f"""
        {prompt_text}
        
        **IMPORTANT FOR SRT:**
        - You are translating an SRT file.
        - **KEEP** all numeric indexes (1, 2, 3...) and timestamps (00:00:00,000 --> ...) EXACTLY as they are.
        - **ONLY** translate the subtitle text lines.
        - Do not merge lines unless necessary for Burmese grammar, but try to keep the sync.
        - Output pure SRT content only.
        """
    else:
        system_instruction = prompt_text
    
    ai_prompt = f"""
    INSTRUCTIONS:
    {system_instruction}
    
    INPUT TEXT:
    {original_text}
    """
    
    try:
        response = client.models.generate_content(model='gemini-2.0-flash', contents=ai_prompt)
        translated_content = response.text.strip()
        
        # Clean potential markdown
        translated_content = translated_content.replace("```srt", "").replace("```", "").strip()
        
        if not translated_content: return False, "❌ AI returned empty translation."

        with open(output_path, "w", encoding="utf-8") as f: f.write(translated_content)
            
        return True, output_path # Return path instead of text for file sending
    except Exception as e:
        return False, f"Error: {str(e)}"

async def run_chat_gemini(user_id, text):
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
    commands = [
        BotCommand("start", "🏠 Home Menu"),
        BotCommand("settings", "⚙️ Configure"),
        BotCommand("translate", "🌍 Translate Last File"),
        BotCommand("heygemini", "🤖 Chat Mode"),
        BotCommand("clearall", "🧹 Clear Data")
    ]
    await application.bot.set_my_commands(commands)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = get_user_state(user_id)
    
    keyboard = [
        [InlineKeyboardButton(f"🎙️ Engine: {state['transcribe_engine'].title()}", callback_data="toggle_transcribe")],
        [InlineKeyboardButton(f"📄 Format: {state['output_format'].upper()}", callback_data="toggle_format")],
        [InlineKeyboardButton("⚙️ Prompt Settings", callback_data="menu_settings")]
    ]
    
    await update.message.reply_text(
        "👋 **Video AI Studio**\n\n"
        "1️⃣ **Send Video/Audio** → Extracts & Transcribes (SRT/TXT)\n"
        "2️⃣ **Send .SRT or .TXT** → Loads file for translation\n"
        "3️⃣ **Type `/translate`** → Translates loaded file\n\n"
        f"**Current Settings:**\nEngine: {state['transcribe_engine']}\nFormat: {state['output_format']}",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📝 View Prompts", callback_data="st_view")],
        [InlineKeyboardButton("✏️ Edit Burmese", callback_data="st_edit_burmese")],
        [InlineKeyboardButton("✏️ Edit Rephrase", callback_data="st_edit_rephrase")],
        [InlineKeyboardButton("🔄 Reset Defaults", callback_data="st_reset")],
        [InlineKeyboardButton("🔙 Back", callback_data="st_back")]
    ]
    await update.message.reply_text("⚙️ **Prompt Settings**", reply_markup=InlineKeyboardMarkup(keyboard))

async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_modes[user_id] = None 
    
    # Check what exists
    p = get_paths(user_id)
    files_found = []
    if os.path.exists(p['srt']): files_found.append("SRT")
    if os.path.exists(p['txt']): files_found.append("TXT")
    
    if not files_found:
        await update.message.reply_text("❌ **No file found.**\nPlease upload a video, audio, srt, or txt file first.")
        return

    keyboard = [
        [InlineKeyboardButton("🇲🇲 To Burmese", callback_data="trans_burmese")],
        [InlineKeyboardButton("🇺🇸 Rephrase English", callback_data="trans_rephrase")],
        [InlineKeyboardButton("✍️ Custom Prompt", callback_data="trans_custom")]
    ]
    
    await update.message.reply_text(
        f"🌍 **Translating...**\nFound: {', '.join(files_found)}",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def perform_translation_logic(update, context, user_id, prompt):
    msg = update.effective_message
    
    status = await msg.reply_text(f"🌍 **Translating...**")
    
    success, result = await run_translate(user_id, prompt)
    
    if success:
        # result is the path to the file
        await context.bot.send_document(msg.chat_id, document=open(result, "rb"), caption="✅ Translated File")
        await status.delete()
        
        # Feedback
        keyboard = [[InlineKeyboardButton("✅ Good", callback_data="feedback_yes"), InlineKeyboardButton("❌ Bad", callback_data="feedback_no")]]
        await context.bot.send_message(chat_id=msg.chat_id, text="Translation OK?", reply_markup=InlineKeyboardMarkup(keyboard))
    else:
        # result is error message
        await status.edit_text(result)

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    state = get_user_state(user_id)
    data = query.data
    
    # --- TOGGLES ---
    if data == "toggle_transcribe":
        state['transcribe_engine'] = "gemini" if state['transcribe_engine'] == "whisper" else "whisper"
        await refresh_start_menu(query, state)

    elif data == "toggle_format":
        state['output_format'] = "txt" if state['output_format'] == "srt" else "srt"
        await refresh_start_menu(query, state)

    # --- SETTINGS & PROMPTS ---
    elif data == "menu_settings" or data == "st_back":
        # Same settings menu as before
        keyboard = [
            [InlineKeyboardButton("📝 View Prompts", callback_data="st_view")],
            [InlineKeyboardButton("✏️ Edit Burmese", callback_data="st_edit_burmese")],
            [InlineKeyboardButton("✏️ Edit Rephrase", callback_data="st_edit_rephrase")],
            [InlineKeyboardButton("🔄 Reset Defaults", callback_data="st_reset")]
        ]
        await query.message.edit_text("⚙️ **Prompt Settings**", reply_markup=InlineKeyboardMarkup(keyboard))

    elif data == "st_view":
        b = get_active_prompt(user_id, "burmese")
        r = get_active_prompt(user_id, "rephrase")
        await send_copyable_message(query.message.chat_id, context.bot, f"🇲🇲 **Burmese:**\n{b}")
        await send_copyable_message(query.message.chat_id, context.bot, f"🇺🇸 **Rephrase:**\n{r}")
        await query.answer()

    elif data == "st_reset":
        state['custom_prompts'] = {}
        await query.answer("Reset!", show_alert=True)

    elif data == "st_edit_burmese":
        user_modes[user_id] = "edit_prompt_burmese"
        await query.message.edit_text("✍️ Send new **Burmese** prompt:")
    
    elif data == "st_edit_rephrase":
        user_modes[user_id] = "edit_prompt_rephrase"
        await query.message.edit_text("✍️ Send new **Rephrase** prompt:")

    # --- TRANSLATE ACTIONS ---
    elif data.startswith("trans_"):
        mode = data.split("_")[1]
        if mode == "custom":
            user_modes[user_id] = "translate_prompt"
            await query.message.reply_text("✍️ Enter custom instruction:")
        else:
            prompt = get_active_prompt(user_id, mode)
            await perform_translation_logic(update, context, user_id, prompt)

    elif data == "feedback_yes":
        await query.message.edit_text("✅ Thanks!")
    elif data == "feedback_no":
        user_modes[user_id] = "translate_prompt"
        await query.message.edit_text("✍️ Tell me how to fix it:")

async def refresh_start_menu(query, state):
    keyboard = [
        [InlineKeyboardButton(f"🎙️ Engine: {state['transcribe_engine'].title()}", callback_data="toggle_transcribe")],
        [InlineKeyboardButton(f"📄 Format: {state['output_format'].upper()}", callback_data="toggle_format")],
        [InlineKeyboardButton("⚙️ Prompt Settings", callback_data="menu_settings")]
    ]
    await query.message.edit_text(
        f"👋 **Video AI Studio**\n\nSettings Updated:\nEngine: {state['transcribe_engine']}\nFormat: {state['output_format']}",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user_id = msg.from_user.id
    text = msg.text
    p = get_paths(user_id)
    state = get_user_state(user_id)
    mode = user_modes.get(user_id)
    
    if text.startswith("/cancel"):
        user_modes[user_id] = None
        await msg.reply_text("❌ Cancelled.")
        return

    if mode == "chat_gemini":
        await context.bot.send_chat_action(chat_id=msg.chat_id, action="typing")
        response = await run_chat_gemini(user_id, text)
        await send_copyable_message(msg.chat_id, context.bot, response)
        return

    if mode and mode.startswith("edit_prompt_"):
        key = mode.replace("edit_prompt_", "")
        if "custom_prompts" not in state: state['custom_prompts'] = {}
        state['custom_prompts'][key] = text
        user_modes[user_id] = None
        await msg.reply_text(f"✅ **{key.title()} Prompt Updated!**")
        return

    if mode == "translate_prompt":
        user_modes[user_id] = None
        await perform_translation_logic(update, context, user_id, text)
        return

    if "http" in text:
        await process_media_logic(update, context, is_url=True)
        return
        
    # Save generic text
    if len(text) > 10:
        with open(p['txt'], "w", encoding="utf-8") as f: f.write(text)
        await msg.reply_text("✅ Text saved as .txt! Type `/translate`.")

async def file_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user_id = msg.from_user.id
    p = get_paths(user_id)
    
    file_obj = await (msg.document or msg.video or msg.audio).get_file()
    name = msg.document.file_name if msg.document else "media.mp4"
    
    # Handle SRT Upload
    if name.lower().endswith('.srt'):
        await msg.reply_text(f"⬇️ **Saving SRT: {name}...**")
        await file_obj.download_to_drive(p['srt'])
        # Also delete old txt to avoid confusion
        if os.path.exists(p['txt']): os.remove(p['txt'])
        await msg.reply_text(f"✅ **SRT Loaded!**\nType `/translate` to convert.")
        return

    # Handle TXT Upload
    if name.lower().endswith('.txt'):
        await msg.reply_text(f"⬇️ **Saving TXT: {name}...**")
        await file_obj.download_to_drive(p['txt'])
        if os.path.exists(p['srt']): os.remove(p['srt'])
        await msg.reply_text(f"✅ **TXT Loaded!**\nType `/translate`.")
        return
        
    await process_media_logic(update, context, is_url=False)

async def process_media_logic(update, context, is_url):
    msg = update.message
    user_id = msg.from_user.id
    p = get_paths(user_id)
    state = get_user_state(user_id)
    fmt = state['output_format'] # 'srt' or 'txt'
    
    status = await msg.reply_text("⏳ **Downloading & Processing...**")
    try:
        clean_temp(user_id)
        
        # 1. Download/Extract Audio
        if is_url:
            cmd = f"yt-dlp --no-check-certificate -f 'bestaudio/best' -x --audio-format mp3 -o '{p['audio']}' {msg.text}"
            subprocess.run(cmd, shell=True)
        else:
            file_obj = await (msg.video or msg.document or msg.audio).get_file()
            await file_obj.download_to_drive(p['input'])
            subprocess.run(f"ffmpeg -y -i {p['input']} -vn -acodec libmp3lame -q:a 2 {p['audio']}", shell=True)
            
        if not os.path.exists(p['audio']): raise Exception("Audio extraction failed.")
        
        await status.edit_text(f"🎙️ **Transcribing ({state['transcribe_engine']} → {fmt.upper()})...**")
        loop = asyncio.get_event_loop()
        
        # 2. Transcribe
        target_file = p['srt'] if fmt == 'srt' else p['txt']
        
        if state['transcribe_engine'] == "gemini":
            engine_name = await loop.run_in_executor(None, run_gemini_transcribe, p['audio'], target_file, fmt)
        else:
            # Whisper always generates both, so we just return the name
            engine_name = await loop.run_in_executor(None, run_whisper, p['audio'], p['srt'], p['txt'])
            
        # 3. Send Result
        if os.path.exists(target_file) and os.path.getsize(target_file) > 0:
            await context.bot.send_document(
                msg.chat_id, 
                document=open(target_file, "rb"), 
                caption=f"📄 Transcript ({fmt.upper()})\nEngine: {engine_name}"
            )
            # Remove the other format to keep state clean for translation
            other = p['txt'] if fmt == 'srt' else p['srt']
            if os.path.exists(other): os.remove(other)

        await status.edit_text("✅ **Finished!**\nType `/translate` to process this file.")

    except Exception as e:
        await status.edit_text(f"❌ Error: {str(e)}")

# ... (Standard Boilerplate: clearall, cancel, main block remain similar)
async def clearall_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    wipe_user_data(update.effective_user.id)
    await update.message.reply_text("🧹 **Cleared!**")

async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_modes[update.effective_user.id] = None
    await update.message.reply_text("✅ Cancelled.")
    
async def heygemini_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_modes[update.effective_user.id] = "chat_gemini"
    await update.message.reply_text("🤖 **Gemini Chat ON**")

if __name__ == '__main__':
    print("🚀 Video AI Bot Running...")
    app = ApplicationBuilder().token(TG_TOKEN).post_init(post_init).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("settings", settings_command))
    app.add_handler(CommandHandler("heygemini", heygemini_command))
    app.add_handler(CommandHandler("translate", translate_command))
    app.add_handler(CommandHandler("clearall", clearall_command))
    app.add_handler(CommandHandler("cancel", cancel_command))
    
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), text_handler))
    app.add_handler(MessageHandler(filters.VIDEO | filters.Document.ALL | filters.AUDIO, file_handler))
    
    app.run_polling()
