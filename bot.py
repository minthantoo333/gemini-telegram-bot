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

# --- 📝 PROMPTS & RULES ---

# Core Burmese Style Rules (Applied to both TXT and SRT)
BURMESE_STYLE = """
Role: Professional Video Narrator & Translator (Burmese).
Strict Translation Guidelines:
1. **Style:** Translate as a **Natural Narrator**. Professional, clear, and engaging.
2. **Multi-Language:** Translate EVERYTHING (English, Korean, etc.) into natural Burmese.
3. **No Brackets:** NEVER keep original words in brackets. (e.g., Use စီအီးအို, not စီအီးအို (CEO)).
4. **Abbreviations:** Transliterate phonetically (e.g., VIP → ဗီအိုင်ပီ, FBI → အက်ဖ်ဘီအိုင်).
5. **Forbidden:** Do NOT use the word 'ပေါ့' (pout). Use professional sentence endings.
"""

# Wrapper for SRT Files to protect timestamps
SRT_RULES = """
**FORMATTING INSTRUCTIONS (STRICT):**
1. The input is an **SRT Subtitle File**.
2. **OUTPUT FORMAT:** You MUST return a valid SRT file.
3. **TIMESTAMPS:** Do NOT change, shift, or remove any timestamps. They must remain exactly as they are in the source.
4. **SEQUENCE NUMBERS:** Preserve the exact sequence numbers (1, 2, 3...).
5. **CONTENT:** Only translate the text content inside the subtitle blocks using the Style Guidelines above.
6. **NO EXTRA TEXT:** Do not add "Here is the translation". Just output the raw SRT data.
"""

DEFAULT_PROMPTS = {
    "burmese": BURMESE_STYLE,
    "rephrase": "Rephrase this English text to be more clear, natural, and reliable."
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
            "transcribe_engine": "whisper", # Default to Whisper for SRT capability
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
        "trans_result": f"downloads/{user_id}_translated" # Extension added dynamically
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
        # Use float32 on CPU to avoid warnings, float16 on GPU
        compute_type = "float16" if device == "cuda" else "int8"
        
        model = WhisperModel("small", device=device, compute_type=compute_type)
        segments, _ = model.transcribe(audio_path, beam_size=5)
        
        with open(srt_path, "w", encoding="utf-8") as srt, open(txt_path, "w", encoding="utf-8") as txt:
            for i, segment in enumerate(segments, start=1):
                start = format_timestamp(segment.start)
                end = format_timestamp(segment.end)
                text = segment.text.strip()
                # Write SRT
                srt.write(f"{i}\n{start} --> {end}\n{text}\n\n")
                # Write TXT
                txt.write(f"{text} ")
        return "Whisper (Local)"
    except Exception as e:
        print(f"Whisper Error: {e}")
        return "Error"

def run_gemini_transcribe(audio_path, srt_path, txt_path):
    print(f"✨ [Gemini] Listening to {audio_path}...")
    try:
        client = genai.Client(api_key=GEMINI_KEY)
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
            
        prompt = "Transcribe this audio into a clean, readable text transcript. Do not use timestamps."
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
        txt_content = response.text.strip()
        with open(txt_path, "w", encoding="utf-8") as f: f.write(txt_content)
        # Delete SRT if it exists from previous run so translation doesn't use old SRT
        if os.path.exists(srt_path): os.remove(srt_path)
        
        return "Gemini 2.0 Flash"
    except Exception as e:
        print(f"Gemini Error: {e}")
        return "Error"

def format_timestamp(seconds):
    import math
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{math.floor(seconds):02},{milliseconds:03}"

# --- 🧠 TRANSLATION ---

async def run_translate(user_id, prompt_text):
    p = get_paths(user_id)
    
    # 1. Determine Source (Prioritize SRT for subtitles)
    source_path = None
    is_srt = False
    
    if os.path.exists(p['srt']):
        source_path = p['srt']
        is_srt = True
    elif os.path.exists(p['txt']):
        source_path = p['txt']
        is_srt = False
    
    if not source_path: return False, "❌ No content found to translate.", None

    print(f"🌍 Translating {source_path} (Is SRT: {is_srt})...")
    client = genai.Client(api_key=GEMINI_KEY)
    
    with open(source_path, "r", encoding="utf-8") as f: original_text = f.read()
    
    # 2. Construct Prompt based on file type
    if is_srt:
        # Combined Prompt: User Style + Strict SRT Rules
        ai_prompt = f"""
        {SRT_RULES}
        
        {prompt_text}
        
        **INPUT SRT DATA:**
        {original_text}
        """
        output_ext = ".srt"
    else:
        # Standard Text Prompt
        ai_prompt = f"""
        User Instruction: "{prompt_text}"
        
        Input Text:
        {original_text}
        """
        output_ext = ".txt"
    
    # 3. Call AI
    try:
        response = client.models.generate_content(model='gemini-2.0-flash', contents=ai_prompt)
        translated_content = response.text.strip()
        
        if not translated_content: return False, "❌ AI returned empty translation.", None

        # Clean up markdown code blocks if AI added them
        if translated_content.startswith("```"):
            translated_content = translated_content.replace("```srt", "").replace("```", "").strip()

        final_path = p['trans_result'] + output_ext
        with open(final_path, "w", encoding="utf-8") as f: f.write(translated_content)
            
        return True, translated_content, final_path
    except Exception as e:
        return False, f"Error: {str(e)}", None

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
        BotCommand("settings", "⚙️ Prompt Settings"),
        BotCommand("translate", "🌍 Translate File"),
        BotCommand("heygemini", "🤖 Chat with Gemini"),
        BotCommand("clearall", "🧹 Clear History"),
        BotCommand("cancel", "❌ Cancel")
    ]
    await application.bot.set_my_commands(commands)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = get_user_state(user_id)
    
    keyboard = [
        [InlineKeyboardButton(f"🎙️ Engine: {state['transcribe_engine'].title()}", callback_data="toggle_transcribe")],
        [InlineKeyboardButton("⚙️ Prompt Settings", callback_data="menu_settings")]
    ]
    
    await update.message.reply_text(
        "👋 **Video AI Studio**\n\n"
        "1️⃣ **Send Video/Link** → Extract & Transcribe\n"
        "2️⃣ **Send .TXT/.SRT** → Load file\n"
        "3️⃣ **Type `/translate`** → Translate\n\n"
        "🗑️ **`/clearall`** to reset.",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def clearall_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    wipe_user_data(user_id)
    await update.message.reply_text("🧹 **All files and history cleared!**")

async def heygemini_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_modes[user_id] = "chat_gemini"
    await update.message.reply_text("🤖 **Gemini Chat Mode ON**\n\nI am listening. Talk to me!\n(Type `/cancel` to exit)")

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
    
    # Check what files are available to inform user
    p = get_paths(user_id)
    file_info = ""
    if os.path.exists(p['srt']): file_info = "\nDetected: 🎬 SRT Subtitle"
    elif os.path.exists(p['txt']): file_info = "\nDetected: 📄 Text File"
    
    keyboard = [
        [InlineKeyboardButton("🇲🇲 To Burmese", callback_data="trans_burmese")],
        [InlineKeyboardButton("🇺🇸 Rephrase English", callback_data="trans_rephrase")],
        [InlineKeyboardButton("✍️ Custom Prompt", callback_data="trans_custom")]
    ]
    
    await update.message.reply_text(
        f"🌍 **Select Translation Action:**{file_info}",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def perform_translation_logic(update, context, user_id, prompt):
    msg = update.effective_message
    
    status = await msg.reply_text(f"🌍 **Translating...**")
    
    success, result_text, final_path = await run_translate(user_id, prompt)
    
    if success and final_path:
        await context.bot.send_document(msg.chat_id, document=open(final_path, "rb"), caption="✅ Translated File")
        await status.delete()
        
        # Feedback
        keyboard = [
            [InlineKeyboardButton("✅ Yes", callback_data="feedback_yes"), InlineKeyboardButton("❌ No", callback_data="feedback_no")]
        ]
        await context.bot.send_message(
            chat_id=msg.chat_id,
            text="Translation ကိုကြိုက်ပါသလား? (Do you like it?)",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    else:
        await status.edit_text(result_text)

async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_modes[user_id] = None
    await update.message.reply_text("✅ Mode exited. Back to normal.")

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    state = get_user_state(user_id)
    data = query.data
    
    if data == "toggle_transcribe":
        new_engine = "gemini" if state['transcribe_engine'] == "whisper" else "whisper"
        state['transcribe_engine'] = new_engine
        keyboard = [[InlineKeyboardButton(f"🎙️ Engine: {new_engine.title()}", callback_data="toggle_transcribe")]]
        await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup(keyboard))

    elif data == "menu_settings" or data == "st_back":
        keyboard = [
            [InlineKeyboardButton("📝 View Prompts", callback_data="st_view")],
            [InlineKeyboardButton("✏️ Edit Burmese", callback_data="st_edit_burmese")],
            [InlineKeyboardButton("✏️ Edit Rephrase", callback_data="st_edit_rephrase")],
            [InlineKeyboardButton("🔄 Reset Defaults", callback_data="st_reset")]
        ]
        await query.message.edit_text("⚙️ **Prompt Settings**\nSelect an option:", reply_markup=InlineKeyboardMarkup(keyboard))

    elif data == "st_view":
        burmese = get_active_prompt(user_id, "burmese")
        rephrase = get_active_prompt(user_id, "rephrase")
        await send_copyable_message(query.message.chat_id, context.bot, f"🇲🇲 **Burmese Rules:**\n{burmese}")
        await send_copyable_message(query.message.chat_id, context.bot, f"🇺🇸 **Rephrase Rules:**\n{rephrase}")
        await query.answer("Prompts sent above!")

    elif data == "st_reset":
        state['custom_prompts'] = {}
        await query.answer("✅ Prompts reset to default!", show_alert=True)

    elif data == "st_edit_burmese":
        user_modes[user_id] = "edit_prompt_burmese"
        await query.message.edit_text("✍️ **Send me the new prompt for BURMESE translation:**\n(Type `/cancel` to stop)")
    
    elif data == "st_edit_rephrase":
        user_modes[user_id] = "edit_prompt_rephrase"
        await query.message.edit_text("✍️ **Send me the new prompt for REPHRASE translation:**\n(Type `/cancel` to stop)")

    elif data == "trans_burmese":
        prompt = get_active_prompt(user_id, "burmese")
        await perform_translation_logic(update, context, user_id, prompt)
    
    elif data == "trans_rephrase":
        prompt = get_active_prompt(user_id, "rephrase")
        await perform_translation_logic(update, context, user_id, prompt)
        
    elif data == "trans_custom":
        user_modes[user_id] = "translate_prompt"
        await query.message.reply_text("✍️ **Enter your custom prompt:**")
    
    elif data == "feedback_yes":
        await query.message.edit_text("ကျေးဇူးတင်ပါတယ်! (Thanks!) ✅")
        
    elif data == "feedback_no":
        user_modes[user_id] = "translate_prompt"
        await query.message.edit_text("✍️ **ဘယ်လို ပြင်ဆင်ချင်ပါသလဲ? (Please type your custom prompt):**")

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user_id = msg.from_user.id
    text = msg.text
    p = get_paths(user_id)
    state = get_user_state(user_id)
    mode = user_modes.get(user_id)
    
    if text.startswith("/cancel"):
        user_modes[user_id] = None
        await msg.reply_text("❌ Action Cancelled.")
        return

    if mode == "chat_gemini":
        await context.bot.send_chat_action(chat_id=msg.chat_id, action="typing")
        response = await run_chat_gemini(user_id, text)
        await send_copyable_message(msg.chat_id, context.bot, response)
        return

    if mode == "edit_prompt_burmese":
        if "custom_prompts" not in state: state['custom_prompts'] = {}
        state['custom_prompts']['burmese'] = text
        user_modes[user_id] = None
        await msg.reply_text("✅ **Burmese Prompt Updated!**")
        return

    if mode == "edit_prompt_rephrase":
        if "custom_prompts" not in state: state['custom_prompts'] = {}
        state['custom_prompts']['rephrase'] = text
        user_modes[user_id] = None
        await msg.reply_text("✅ **Rephrase Prompt Updated!**")
        return

    if mode == "translate_prompt":
        user_modes[user_id] = None
        await perform_translation_logic(update, context, user_id, text)
        return

    if "http" in text:
        await process_video_logic(update, context, is_url=True)
        return
        
    if len(text) > 10:
        with open(p['txt'], "w", encoding="utf-8") as f: f.write(text)
        await msg.reply_text("✅ **Text saved!**\nType `/translate`.")
        return

async def file_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user_id = msg.from_user.id
    p = get_paths(user_id)
    
    file_obj = await (msg.document or msg.video or msg.audio).get_file()
    name = msg.document.file_name if msg.document else "video.mp4"
    
    if name.lower().endswith('.srt'):
        await msg.reply_text(f"⬇️ **Saving Subtitle {name}...**")
        await file_obj.download_to_drive(p['srt'])
        await msg.reply_text(f"✅ **SRT Loaded!**\nType `/translate` to translate subtitles.")
        return

    if name.lower().endswith('.txt'):
        await msg.reply_text(f"⬇️ **Saving Text {name}...**")
        await file_obj.download_to_drive(p['txt'])
        await msg.reply_text(f"✅ **Text Loaded!**\nType `/translate`.")
        return
        
    await process_video_logic(update, context, is_url=False)

async def process_video_logic(update, context, is_url):
    msg = update.message
    user_id = msg.from_user.id
    p = get_paths(user_id)
    state = get_user_state(user_id)
    
    status = await msg.reply_text("⏳ **Downloading & Processing...**")
    try:
        clean_temp(user_id)
        
        # Audio Extraction
        if is_url:
            cmd = f"yt-dlp --no-check-certificate -f 'bestaudio/best' -x --audio-format mp3 -o '{p['audio']}' {msg.text}"
            subprocess.run(cmd, shell=True)
        else:
            file_obj = await (msg.video or msg.document or msg.audio).get_file()
            await file_obj.download_to_drive(p['input'])
            subprocess.run(f"ffmpeg -y -i {p['input']} -vn -acodec libmp3lame -q:a 2 {p['audio']}", shell=True)
            if os.path.exists(p['input']): os.remove(p['input'])
            
        if not os.path.exists(p['audio']): raise Exception("Audio extraction failed.")
        
        await status.edit_text(f"🎙️ **Transcribing ({state['transcribe_engine']})...**")
        loop = asyncio.get_event_loop()
        
        # Transcribe
        if state['transcribe_engine'] == "gemini":
            engine_name = await loop.run_in_executor(None, run_gemini_transcribe, p['audio'], p['srt'], p['txt'])
        else:
            engine_name = await loop.run_in_executor(None, run_whisper, p['audio'], p['srt'], p['txt'])
            
        # Send Results
        await status.delete()
        
        if os.path.exists(p['txt']) and os.path.getsize(p['txt']) > 0:
            await context.bot.send_document(msg.chat_id, document=open(p['txt'], "rb"), caption=f"📄 Transcript ({engine_name})")
        
        # KEY CHANGE: If using Whisper, also send SRT
        if os.path.exists(p['srt']) and state['transcribe_engine'] == "whisper":
            await context.bot.send_document(msg.chat_id, document=open(p['srt'], "rb"), caption=f"🎬 Subtitles ({engine_name})")

        await context.bot.send_message(msg.chat_id, text="✅ **Finished!**\nType `/translate` to select action.")

    except Exception as e:
        await status.edit_text(f"❌ Error: {str(e)}")

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
