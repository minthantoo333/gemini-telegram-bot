import TelegramBot from "node-telegram-bot-api";
import { GoogleGenerativeAI } from "@google/generative-ai";

const TELEGRAM_TOKEN = process.env.TELEGRAM_TOKEN;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

// Telegram bot
const bot = new TelegramBot(TELEGRAM_TOKEN, { polling: true });

// Gemini
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

bot.on("message", async (msg) => {
  const chatId = msg.chat.id;
  const userText = msg.text || "";

  try {
    const result = await model.generateContent(userText);
    const reply = result.response.text();
    bot.sendMessage(chatId, reply);
  } catch (e) {
    bot.sendMessage(chatId, "Error: " + e.message);
  }
});

console.log("Bot is running...");
