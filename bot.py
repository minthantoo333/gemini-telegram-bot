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
# --- ⚙️ CONFIGURATION ---
# We use os.getenv to read from the system securely
TG_TOKEN = os.getenv("TG_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_KEY")

# Check if keys are missing
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
   - **Korean:** Translate Korean dialogue naturally into Burmese context.
3. **No Brackets/Parentheses:** **NEVER** keep the original English word in brackets.
   - ❌ Bad: စီအီးအို (CEO)
   - ✅ Good: စီအီးအို
4. **Abbreviations:** Transliterate phonetically (e.g., VIP → ဗီအိုင်ပီ, FBI → အက်ဖ်ဘီအိုင်).
5. **Forbidden:** Do NOT use the word 'ပေါ့' (pout). Use professional sentence endings.
6. **Output:** Return ONLY the Burmese translation.
""",
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
            "transcribe_engine": "gemini",
            "custom_prompts": {} # Stores user overrides
        }
    return user_prefs[user_id]

def get_active_prompt(user_id, key):
    """Returns the user's custom prompt if it exists, otherwise the default."""
    state = get_user_state(user_id)
    custom = state.get("custom_prompts", {}).get(key)
    return custom if custom else DEFAULT_PROMPTS[key]

def get_paths(user_id):
    return {
        "input": f"downloads/{user_id}_input.mp4",
        "audio": f"downloads/{user_id}_audio.mp3",
        "srt": f"downloads/{user_id}_subs.srt",
        "txt": f"downloads/{user_id}_transcript.txt",
        "trans_txt": f"downloads/{user_id}_translated.txt"
    }

def clean_temp(user_id):
    p = get_paths(user_id)
    if os.path.exists(p['input']): os.remove(p['input'])

def wipe_user_data(user_id):
    for f in glob.glob(f"downloads/{user_id}_*"):
        try: os.remove(f)
        except: pass
    for f in glob.glob(f"temp/{user_id}_*"):
        try: os.remove(f)
        except: pass
    
    if user_id in user_prefs: del user_prefs[user_id]
    if user_id in user_modes: del user_modes[user_id]
    if user_id in chat_histories: del chat_histories[user_id]

async def send_copyable_message(chat_id, bot, text):
    """Sends text in markdown code blocks for one-click copying."""
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
        with open(txt_path, "w") as f: f.write("[Error]")
        return "Error"

def run_gemini_transcribe(audio_path, srt_path, txt_path):
    print(f"✨ [Gemini] Listening to {audio_path}...")
    try:
        client = genai.Client(api_key=GEMINI_KEY)
        
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
            
        prompt = "Transcribe this audio into a clean, readable text transcript. Do not use timestamps. Just the text."
        
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
        if not txt_content: txt_content = "[Silence or No Speech Detected]"
            
        with open(txt_path, "w", encoding="utf-8") as f: f.write(txt_content)
        # Dummy SRT
        with open(srt_path, "w", encoding="utf-8") as f: f.write("1\n00:00:00,000 --> 00:00:05,000\n[Transcript Only]")
        
        return "Gemini 2.0 Flash"
        
    except Exception as e:
        print(f"Gemini Error: {e}")
        with open(txt_path, "w") as f: f.write("[Error]")
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
    
    source_path = None
    if os.path.exists(p['txt']): source_path = p['txt']
    elif os.path.exists(p['srt']): source_path = p['srt']
    
    if not source_path: return False, "❌ No content found to translate."

    print(f"🌍 Translating {source_path}...")
    client = genai.Client(api_key=GEMINI_KEY)
    
    with open(source_path, "r", encoding="utf-8") as f: original_text = f.read()
    
    # Construct Final Prompt
    ai_prompt = f"""
    User Instruction: "{prompt_text}"
    
    Input Text:
    {original_text}
    """
    
    try:
        response = client.models.generate_content(model='gemini-2.0-flash', contents=ai_prompt)
        translated_content = response.text.strip()
        
        if not translated_content: return False, "❌ AI returned empty translation."

        with open(p['trans_txt'], "w", encoding="utf-8") as f: f.write(translated_content)
            
        return True, translated_content
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
        "2️⃣ **Send .TXT** → Load file\n"
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
    
    keyboard = [
        [InlineKeyboardButton("🇲🇲 To Burmese", callback_data="trans_burmese")],
        [InlineKeyboardButton("🇺🇸 Rephrase English", callback_data="trans_rephrase")],
        [InlineKeyboardButton("✍️ Custom Prompt", callback_data="trans_custom")]
    ]
    
    await update.message.reply_text(
        "🌍 **Select Translation Action:**\n(Result: .txt file)",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def perform_translation_logic(update, context, user_id, prompt):
    msg = update.effective_message
    p = get_paths(user_id)
    
    status = await msg.reply_text(f"🌍 **Translating...**")
    
    success, result_text = await run_translate(user_id, prompt)
    
    if success:
        if os.path.exists(p['trans_txt']):
            await context.bot.send_document(msg.chat_id, document=open(p['trans_txt'], "rb"), caption="📄 Translated File")
        
        await status.delete()
        
        # --- ASK FEEDBACK ---
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
    
    # --- TRANSCRIBE TOGGLE ---
    if data == "toggle_transcribe":
        new_engine = "gemini" if state['transcribe_engine'] == "whisper" else "whisper"
        state['transcribe_engine'] = new_engine
        keyboard = [[InlineKeyboardButton(f"🎙️ Engine: {new_engine.title()}", callback_data="toggle_transcribe")]]
        await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup(keyboard))

    # --- SETTINGS MENU ---
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
        
        await send_copyable_message(query.message.chat_id, context.bot, f"🇲🇲 **Burmese Prompt:**\n{burmese}")
        await send_copyable_message(query.message.chat_id, context.bot, f"🇺🇸 **Rephrase Prompt:**\n{rephrase}")
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

    # --- TRANSLATION ACTIONS ---
    elif data == "trans_burmese":
        prompt = get_active_prompt(user_id, "burmese")
        await perform_translation_logic(update, context, user_id, prompt)
    
    elif data == "trans_rephrase":
        prompt = get_active_prompt(user_id, "rephrase")
        await perform_translation_logic(update, context, user_id, prompt)
        
    elif data == "trans_custom":
        user_modes[user_id] = "translate_prompt"
        await query.message.reply_text("✍️ **Enter your custom prompt:**")
    
    # --- FEEDBACK ---
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
    
    # --- CANCEL CHECK ---
    if text.startswith("/cancel"):
        user_modes[user_id] = None
        await msg.reply_text("❌ Action Cancelled.")
        return

    # --- CHAT MODE ---
    if mode == "chat_gemini":
        await context.bot.send_chat_action(chat_id=msg.chat_id, action="typing")
        response = await run_chat_gemini(user_id, text)
        await send_copyable_message(msg.chat_id, context.bot, response)
        return

    # --- SETTINGS: EDIT PROMPTS ---
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

    # --- TRANSLATION CUSTOM ---
    if mode == "translate_prompt":
        user_modes[user_id] = None
        await perform_translation_logic(update, context, user_id, text)
        return

    # --- URL HANDLING ---
    if "http" in text:
        await process_video_logic(update, context, is_url=True)
        return
        
    # --- SAVE TEXT FILE ---
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
    
    if name.lower().endswith('.txt') or name.lower().endswith('.srt'):
        target = p['txt']
        await msg.reply_text(f"⬇️ **Saving {name}...**")
        await file_obj.download_to_drive(target)
        await msg.reply_text(f"✅ **Loaded {name}!**\nType `/translate`.")
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
        
        if state['transcribe_engine'] == "gemini":
            engine_name = await loop.run_in_executor(None, run_gemini_transcribe, p['audio'], p['srt'], p['txt'])
        else:
            engine_name = await loop.run_in_executor(None, run_whisper, p['audio'], p['srt'], p['txt'])
            
        if os.path.getsize(p['txt']) > 0:
            await context.bot.send_document(msg.chat_id, document=open(p['txt'], "rb"), caption=f"📄 Transcript ({engine_name})")

        await status.edit_text("✅ **Finished!**\nType `/translate` to select action.")

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
