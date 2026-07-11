"""
BAKOME Local AI Studio — AI + Solana Enhancements
DeepSeek AI • Jupiter API • Birdeye • Telegram @merryglann
"""
import os, json, httpx
from typing import Optional, Dict, Any

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-free-key")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
JUPITER_URL = "https://quote-api.jup.ag/v6/quote"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = "@merryglann"

async def deepseek_analyze(prompt: str, system: str = "Expert crypto & blockchain.") -> str:
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "deepseek-chat", "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}], "temperature": 0.7, "max_tokens": 2048}
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(DEEPSEEK_URL, json=payload, headers=headers)
        if resp.status_code == 200: return resp.json()["choices"][0]["message"]["content"]
        return f"Erreur: {resp.status_code}"

async def get_sol_price() -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient() as client:
            params = {"inputMint": "So11111111111111111111111111111111111111112", "outputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", "amount": 1000000000, "slippageBps": 50}
            resp = await client.get(JUPITER_URL, params=params)
            if resp.status_code == 200:
                data = resp.json()
                return {"price_usdc": float(data.get("outAmount", 0)) / 1_000_000, "route": "Jupiter"}
    except: pass
    return {"price_usdc": 0, "route": "unknown"}

async def send_telegram_alert(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN: return False
    try:
        async with httpx.AsyncClient() as client:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            resp = await client.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": f"🔔 BAKOME Studio\n{message}", "parse_mode": "HTML"})
            return resp.status_code == 200
    except: return False

print("✅ AI Enhancements loaded!")
