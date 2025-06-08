# chatbot_module.py
import requests

API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
headers = {
    "Authorization": "Bearer YOUR_HF_TOKEN_HERE"  # Replace with your token
}

def get_bot_response(user_input):
    payload = {
        "inputs": f"<|user|>\n{user_input}\n<|assistant|>",
        "parameters": {"max_new_tokens": 100, "temperature": 0.7},
        "options": {"wait_for_model": True}
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        return f"[Error {response.status_code}]: {response.text}"
    
    try:
        return response.json()[0]["generated_text"].split("<|assistant|>")[-1].strip()
    except Exception as e:
        return f"⚠️ Unexpected format: {e}"
