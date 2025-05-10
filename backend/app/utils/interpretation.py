import requests
import os
import numpy as np
from dotenv import load_dotenv

from app.utils.logger import logger

load_dotenv()
def format_interpretation_prompt(allocation, sentiment_scores, bl_returns, future_prices):
    volatility = {t: round(np.std(p), 4) for t, p in future_prices.items()}

    prompt = (
        "Please interpret this portfolio optimization result:\n\n"
        f"Asset Allocation: {allocation}\n"
        f"Sentiment Scores: {sentiment_scores}\n"
        f"Black–Litterman Expected Returns: {bl_returns}\n"
        f"Predicted Volatility (ANN): {volatility}\n\n"
        "Provide a client-friendly interpretation in bullet points."
    )

    return prompt
def generate_interpretation_openrouter(prompt, model="mistralai/mistral-7b-instruct"):
    # api_key = "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"

    api_key = os.getenv("OPENROUTER_API_KEY")  # store securely or hardcode temporarily
    logger.debug(f"API key = {api_key}")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost",  # or your frontend domain
        "Content-Type": "application/json"
    }

    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a financial advisor creating easy-to-understand interpretations of portfolio optimization results."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"OpenRouter error: {response.text}")